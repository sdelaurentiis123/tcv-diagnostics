#!/usr/bin/env python3
"""Render O1 codec-transport figures from the compact committed result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CODECS = ("f8", "z44")
QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)
COLORS = {"f8": "#0072B2", "z44": "#D55E00"}
DISPLAY = {
    "particle": "Particle",
    "electron_internal_energy": "Electron energy",
    "ion_internal_energy": "Ion energy",
    "total_internal_energy": "Total energy",
}
CADENCE_MICROSECONDS = 3.131905426352636


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("result_type") != "phase2_o1_codec_transport_oracle_compact":
        raise ValueError("input is not the compact O1 codec-transport result")
    if (
        result["scope"].get("run_id") != "85604"
        or result["scope"].get("shot_85606_accessed") is not False
        or result["scope"].get("training_performed") is not False
    ):
        raise ValueError("result does not satisfy the frozen O1 scope")
    if result["scope"].get("zperiod") != 5:
        raise ValueError("result does not use zperiod=5")
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
        "Description": (
            "Paper 0 O1 deterministic codec transport; 85604 only; no training."
        ),
    }
    fig.savefig(output_dir / f"{stem}.svg", metadata=metadata)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, metadata=metadata)
    plt.close(fig)


def authoritative_error(
    result: dict[str, Any],
    codec: str,
    quantity: str,
    reduction: str,
) -> float:
    return float(
        result["codec_results"][codec]["overall"]["comparisons"][
            "P0_vs_R_authoritative"
        ]["quantities"][quantity][reduction]["metrics"]["relative_l2"]
    )


def plot_attribution(
    result: dict[str, Any],
    output_dir: Path,
    result_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    x = np.arange(len(QUANTITIES), dtype=np.float64)
    width = 0.34
    panels = (
        (
            "strict_faces",
            "All strict physical face contributions",
            0.25,
            0.34,
            "Relative L2 error",
        ),
        (
            "separatrix",
            "Integrated confined-separatrix wedge flow",
            0.20,
            0.22,
            "Relative L2 error",
        ),
    )
    for axis, (reduction, title, threshold, upper, ylabel) in zip(axes, panels):
        for codec_index, codec in enumerate(CODECS):
            values = np.asarray(
                [
                    authoritative_error(result, codec, quantity, reduction)
                    for quantity in QUANTITIES
                ]
            )
            positions = x + (codec_index - 0.5) * width
            bars = axis.bar(
                positions,
                values,
                width=width,
                color=COLORS[codec],
                label=codec,
                alpha=0.9,
            )
            axis.bar_label(
                bars,
                labels=[f"{100.0 * value:.1f}%" for value in values],
                padding=2,
                fontsize=8,
                rotation=0,
            )
        axis.axhspan(0.0, threshold, color="#009E73", alpha=0.06, zorder=0)
        axis.axhline(
            threshold,
            color="#009E73",
            linestyle="--",
            linewidth=1.2,
            label=f"frozen gate ≤ {100 * threshold:.0f}%",
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, upper)
        axis.set_xticks(x, [DISPLAY[quantity] for quantity in QUANTITIES])
        axis.tick_params(axis="x", rotation=18)
    axes[0].legend(ncol=3, loc="upper right")
    fig.suptitle(
        "Codec transport fidelity depends on the spatial reduction",
        fontsize=13,
    )
    fig.text(
        0.5,
        -0.025,
        (
            "Relative L2 compares direct-pressure native truth (P0) with each "
            "decoded codec state (R). Face panel: 90,119,952 weighted face points; "
            "surface panel: 624 one-fifth-wedge time samples."
        ),
        ha="center",
        va="top",
        fontsize=8.5,
    )
    save(fig, output_dir, "codec-transport-attribution", result_path)


def plot_surface_time_series(
    result: dict[str, Any],
    output_dir: Path,
    result_path: Path,
) -> None:
    source = result["surface_series_si"]
    truth = source["truth_P0"]
    reconstructions = source["reconstruction_R"]
    frames = np.arange(result["scope"]["frame_count"])
    time_microseconds = frames * CADENCE_MICROSECONDS
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), sharex=True)
    fig.subplots_adjust(top=0.84, bottom=0.11, hspace=0.27, wspace=0.12)
    for axis, quantity in zip(axes.flat, QUANTITIES):
        truth_values = np.asarray(truth[quantity], dtype=np.float64)
        scale = 1.0e17 if quantity == "particle" else 1.0
        axis.plot(
            time_microseconds,
            truth_values / scale,
            color="#222222",
            linewidth=1.15,
            alpha=0.82,
            label="native P0 truth",
            zorder=3,
        )
        for codec in CODECS:
            values = np.asarray(reconstructions[codec][quantity], dtype=np.float64)
            axis.plot(
                time_microseconds,
                values / scale,
                color=COLORS[codec],
                linewidth=0.85,
                alpha=0.9,
                label=f"{codec} reconstruction",
            )
        f8_error = authoritative_error(result, "f8", quantity, "separatrix")
        z44_error = authoritative_error(result, "z44", quantity, "separatrix")
        axis.text(
            0.015,
            0.97,
            f"relative L2: f8 {100*f8_error:.1f}%  ·  z44 {100*z44_error:.1f}%",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
        )
        axis.set_title(DISPLAY[quantity])
        axis.set_ylabel(
            r"Outward wedge flow ($10^{17}\ \mathrm{s}^{-1}$)"
            if quantity == "particle"
            else "Outward wedge flow (W)"
        )
    for axis in axes[-1]:
        axis.set_xlabel("Time from first stored frame (µs)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.91))
    fig.suptitle(
        "Confined-separatrix transport through the simulated one-fifth wedge",
        fontsize=13,
        y=0.975,
    )
    fig.text(
        0.5,
        0.018,
        (
            "No smoothing. Positive is outward from the confined region. Electron, "
            "ion, and total curves are internal-energy flow with the 3/2 factor "
            "applied before SI conversion."
        ),
        ha="center",
        va="top",
        fontsize=8.5,
    )
    save(fig, output_dir, "codec-separatrix-transport-time-series", result_path)


def plot_error_ladder(
    result: dict[str, Any],
    output_dir: Path,
    result_path: Path,
) -> None:
    stages = (
        "P0_vs_P1_state_gap",
        "P1_vs_P2_input_roundtrip",
        "P0_vs_R_authoritative",
    )
    stage_labels = (
        "C5T state\nP0 → P1",
        "88→81 input\nP1 → P2",
        "end-to-end codec\nP0 → R",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.0), sharex=True)
    fig.subplots_adjust(top=0.83, bottom=0.15, hspace=0.28, wspace=0.20)
    for axis, quantity in zip(axes.flat, QUANTITIES):
        for codec in CODECS:
            comparisons = result["codec_results"][codec]["overall"]["comparisons"]
            for reduction, marker, linestyle, label_suffix in (
                ("strict_faces", "o", "-", "faces"),
                ("separatrix", "s", "--", "surface"),
            ):
                values = []
                for stage in stages:
                    value = comparisons[stage]["quantities"][quantity][reduction][
                        "metrics"
                    ]["relative_l2"]
                    values.append(max(float(value), 1.0e-12))
                axis.plot(
                    np.arange(3),
                    values,
                    color=COLORS[codec],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.2,
                    markersize=4.5,
                    label=f"{codec} {label_suffix}",
                )
        axis.set_yscale("log")
        axis.set_ylim(5e-13, 1.0)
        axis.set_title(DISPLAY[quantity])
        axis.set_ylabel("Relative L2 error")
        axis.set_xticks(np.arange(3), stage_labels)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.90))
    fig.suptitle(
        "Transport-error attribution: state and resampling are tiny; codec error dominates",
        fontsize=13,
        y=0.975,
    )
    fig.text(
        0.5,
        0.018,
        (
            "Particle P0→P1 error is exactly zero and is drawn at 10⁻¹² only to "
            "appear on the logarithmic axis. f8 and z44 overlap in the first two "
            "stages because they share the same truth and input paths."
        ),
        ha="center",
        va="top",
        fontsize=8.5,
    )
    save(fig, output_dir, "codec-transport-error-ladder", result_path)


def main() -> None:
    args = parse_args()
    result_path = args.result.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    result = load_result(result_path)
    configure_style()
    plot_attribution(result, output_dir, result_path)
    plot_surface_time_series(result, output_dir, result_path)
    plot_error_ladder(result, output_dir, result_path)
    print(f"wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
