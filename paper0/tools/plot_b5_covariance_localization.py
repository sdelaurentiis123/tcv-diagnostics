#!/usr/bin/env python3
"""Render the frozen B5 covariance-localization figures from compact JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from paper0.tools import localize_b5_covariance as localization


EXPECTED_SHA256 = (
    "331e7f3ff5d221d0d3720d9112ce90436d8330647501a2268f974867bbc140d2"
)
EXPECTED_SCOPE = "B5_read_only_covariance_localization_85604"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_result(path: Path) -> dict[str, Any]:
    if _sha256(path) != EXPECTED_SHA256:
        raise ValueError("covariance-localization result does not match the frozen hash")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("scope") != EXPECTED_SCOPE:
        raise ValueError("input is not the frozen B5 covariance-localization result")
    if result.get("development_run") != "85604":
        raise ValueError("result is not restricted to development simulation 85604")
    if result.get("held_out_85606_read") is not False:
        raise ValueError("result does not certify exclusion of held-out simulation 85606")
    boundaries = result.get("scientific_boundaries", {})
    for key in (
        "model_training_performed",
        "model_inference_performed",
        "assimilation_performed",
        "diagnostic_ranking_performed",
        "O3_launched",
    ):
        if boundaries.get(key) is not False:
            raise ValueError(f"result does not preserve frozen boundary {key}")
    return result


def _plot_spatial_acf(
    output: Path,
    covariance_objects: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Render the executed spatial plot with separate title and legend bands."""
    fig, axes = plt.subplots(3, 5, figsize=(19, 10), sharey=True)
    colors = ("#6b7280", "#111827", "#2563eb", "#dc2626")
    for row, axis_name in enumerate(("x", "y", "stored_toroidal_z")):
        for column, field in enumerate(localization.B5_COVARIANCE_FIELDS):
            ax = axes[row, column]
            for object_name, color in zip(localization.OBJECT_ORDER, colors):
                record = covariance_objects[object_name]["spatial_autocorrelation"][
                    axis_name
                ]
                ax.plot(
                    record["lags_cells"],
                    record["fields"][field]["correlation"],
                    label=localization.OBJECT_LABELS[object_name],
                    color=color,
                    linewidth=1.7,
                )
            ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
            ax.axhline(0.1, color="#d1d5db", linewidth=0.7, linestyle="--")
            ax.axhline(-0.1, color="#d1d5db", linewidth=0.7, linestyle="--")
            ax.set_title(f"{field}: {axis_name}")
            if column == 0:
                ax.set_ylabel("pooled normalized correlation")
            if row == 2:
                ax.set_xlabel("lag (stored grid cells)")
            ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "B5 covariance localization: spatial correlation of residual objects\n"
        "phi is gauge-fixed; realized residuals/innovations are axisymmetric-bias centered",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return localization._save_figure(fig, output, "b5-covariance-spatial-acf")


def _plot_toroidal_power(
    output: Path,
    covariance_objects: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Render the executed toroidal plot with nonoverlapping top matter."""
    objects = (
        "validation_H1_residual",
        "B5_ensemble_anomaly",
        "B5_innovation",
    )
    bands = tuple(localization.B5_COVARIANCE_TOROIDAL_BANDS)
    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    width = 0.24
    positions = np.arange(len(bands))
    for field_index, field in enumerate(localization.B5_COVARIANCE_FIELDS):
        ax = axes[field_index]
        for object_index, object_name in enumerate(objects):
            record = covariance_objects[object_name]["toroidal_support"]["fields"][
                field
            ]["bands"]
            values = [record[band]["power_fraction"] for band in bands]
            ax.bar(
                positions + (object_index - 1) * width,
                values,
                width=width,
                label=localization.OBJECT_LABELS[object_name],
            )
        ax.set_ylabel(f"{field}\npower fraction")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(
        positions,
        [
            "k=0\nn=0",
            "k=1–3\nn=5–15",
            "k=4–5\nn=20–25",
            "k=6–7\nn=30–35",
            "k≥8\nn≥40",
        ],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Parseval-weighted toroidal support (stored k maps to full-torus n=5k)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return localization._save_figure(fig, output, "b5-covariance-toroidal-power")


def write_readout_figures(output: Path, result: Mapping[str, Any]) -> list[str]:
    """Render the six executed diagnostics without recomputing any metric."""
    figures: list[str] = []
    covariance = result["covariance_objects"]
    figures.extend(_plot_spatial_acf(output, covariance))
    figures.extend(
        localization._plot_cross_field(
            output,
            covariance,
            result["dependence_distances"],
        )
    )
    figures.extend(_plot_toroidal_power(output, covariance))
    figures.extend(
        localization._plot_transport_covariance(
            output,
            result["transport_covariance"],
        )
    )
    figures.extend(localization._plot_variogram(output, result["variogram_scores"]))
    figures.extend(localization._plot_history(output, result["history_probe"]))
    return figures


def main() -> None:
    args = parse_args()
    result = load_frozen_result(args.result.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = write_readout_figures(args.output_dir.resolve(), result)
    print(json.dumps({"figure_count": len(names), "figures": names}, sort_keys=True))


if __name__ == "__main__":
    main()
