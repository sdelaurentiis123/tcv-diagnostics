#!/usr/bin/env python3
"""Render physics-first figures for the frozen S0 spatial reconstruction.

The builder reads only the completed old-85604 S0 result directory.  It does
not load a temporal model, raw simulation archive, NERSC data, or 85606.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


INK = "#18232d"
MUTED = "#65717d"
GRID = "#dce3e7"
BLUE = "#3478a8"
TEAL = "#16827f"
GOLD = "#c18a2f"
CORAL = "#c45a4d"
PURPLE = "#7966a5"
FIELD_COLORS = {"Ne": BLUE, "Pe": TEAL, "Pi": GOLD, "phi": PURPLE}
FIELD_LABELS = {"Ne": r"$\delta N_e$", "Pe": r"$\delta P_e$", "Pi": r"$\delta P_i$", "phi": r"$\delta \phi$"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "grid.linewidth": 0.65,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "paper0-s0-20260828",
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str, description: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": stem.replace("_", " ").title(),
        "Description": description,
        "Date": "2026-08-28",
        "Source": "TCV/Hermes 85604 development data; S0 simultaneous reconstruction",
    }
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", metadata=metadata)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=220, metadata=metadata)
    plt.close(fig)


def plot_poloidal(ax: plt.Axes, values: np.ndarray, r: np.ndarray, z: np.ndarray, *, limit: float, title: str) -> None:
    mesh = ax.pcolormesh(r, z, values, shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("major radius R (m)")
    ax.set_ylabel("vertical position Z (m)")
    ax.set_title(title, loc="left")
    ax.grid(False)
    return mesh


def build_hero(result_dir: Path, output_dir: Path) -> None:
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    footprints = json.loads((result_dir / "footprints.json").read_text(encoding="utf-8"))
    with np.load(result_dir / "hero_frame.npz") as data:
        truth = np.asarray(data["truth"])
        ridge = np.asarray(data["ridge"])
        error = np.asarray(data["error"])
        r = np.asarray(data["major_radius_m"])
        zz = np.asarray(data["vertical_position_m"])
        mask = np.asarray(data["strict_operator_mask"], dtype=bool)
        c_truth = np.asarray(data["heldout_c_truth"], dtype=float)
        c_pred = np.asarray(data["heldout_c_prediction"], dtype=float)
        frame = int(np.asarray(data["frame"]).item())
    with np.load(result_dir / "validation_diagnostics.npz") as data:
        frames = np.asarray(data["frames"], dtype=int)
        c_all_truth = np.asarray(data["heldout_c_truth"], dtype=float)
        c_all_pred = np.asarray(data["heldout_c_ridge"], dtype=float)

    z_index = 44
    ne_truth = truth[0, :, :, z_index]
    ne_ridge = ridge[0, :, :, z_index]
    ne_error = error[0, :, :, z_index]
    finite_truth = ne_truth[mask]
    limit = float(np.quantile(np.abs(finite_truth[np.isfinite(finite_truth)]), 0.99))

    fig = plt.figure(figsize=(14.4, 8.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.08, 0.92])
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    mesh = plot_poloidal(axes[0], ne_truth, r, zz, limit=limit, title="A. Hidden truth at held-out toroidal plane")
    plot_poloidal(axes[1], ne_ridge, r, zz, limit=limit, title="B. Reconstructed from diagnostics A + B")
    plot_poloidal(axes[2], ne_error, r, zz, limit=limit, title="C. Reconstruction residual")
    cbar = fig.colorbar(mesh, ax=axes, location="right", shrink=0.83, pad=0.02)
    cbar.set_label(r"train-normalized density fluctuation $\delta N_e$")

    for entry in footprints["kept"]:
        if entry["family"] != "C":
            continue
        ix, iy, iz = entry["center_xyz"]
        if iz == z_index:
            for ax in axes:
                ax.scatter(r[ix, iy], zz[ix, iy], s=30, marker="s", facecolors="none", edgecolors=CORAL, linewidths=1.2)
    axes[0].text(
        0.02,
        0.03,
        "red squares: held-out diagnostic C",
        transform=axes[0].transAxes,
        fontsize=8,
        color=CORAL,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    ax_top = fig.add_subplot(grid[1, 0])
    theta = np.linspace(0.0, 2.0 * np.pi / 5.0, 300)
    radial_span = [float(np.nanmin(r)), float(np.nanmax(r))]
    for radius in radial_span:
        ax_top.plot(radius * np.cos(theta), radius * np.sin(theta), color=GRID, linewidth=1)
    centers = {"A": [], "B": [], "C": []}
    for entry in footprints["kept"]:
        ix, iy, iz = entry["center_xyz"]
        angle = (2.0 * np.pi / 5.0) * (iz / 88.0)
        radius = float(r[ix, iy])
        centers[entry["family"]].append((radius * np.cos(angle), radius * np.sin(angle)))
    family_style = {"A": (BLUE, "observed A"), "B": (TEAL, "observed B"), "C": (CORAL, "held-out C")}
    for family, points in centers.items():
        xy = np.asarray(points)
        color, label = family_style[family]
        ax_top.scatter(xy[:, 0], xy[:, 1], s=42, color=color, label=label, zorder=3)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("top-down x (m)")
    ax_top.set_ylabel("top-down y (m)")
    ax_top.set_title("D. Diagnostic toroidal positions", loc="left")
    ax_top.legend(loc="upper left", fontsize=8)

    ax_trace = fig.add_subplot(grid[1, 1])
    true_mean = c_all_truth.mean(axis=1)
    pred_mean = c_all_pred.mean(axis=1)
    ax_trace.plot(frames, true_mean, color=INK, linewidth=1.8, label="held-out C truth")
    ax_trace.plot(frames, pred_mean, color=TEAL, linewidth=1.5, label="predicted from A + B")
    ax_trace.axvline(frame, color=CORAL, linestyle="--", linewidth=1, label=f"hero frame {frame}")
    ax_trace.set_xlabel("85604 frame")
    ax_trace.set_ylabel(r"mean held-out $\delta N_e$")
    ax_trace.set_title("E. Held-out signal through validation", loc="left")
    ax_trace.legend(fontsize=8)

    ax_scatter = fig.add_subplot(grid[1, 2])
    flat_truth = c_all_truth.ravel()
    flat_pred = c_all_pred.ravel()
    scatter_limit = float(np.quantile(np.abs(np.concatenate([flat_truth, flat_pred])), 0.995))
    ax_scatter.scatter(flat_truth, flat_pred, s=10, alpha=0.3, color=TEAL, edgecolors="none")
    ax_scatter.plot([-scatter_limit, scatter_limit], [-scatter_limit, scatter_limit], color=INK, linewidth=1, linestyle="--")
    ax_scatter.scatter(c_truth, c_pred, s=34, color=CORAL, edgecolors="white", linewidths=0.5, label=f"frame {frame}")
    ax_scatter.set_xlim(-scatter_limit, scatter_limit)
    ax_scatter.set_ylim(-scatter_limit, scatter_limit)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel("true held-out C")
    ax_scatter.set_ylabel("predicted held-out C")
    ax_scatter.set_title("F. Six held-out channels × 128 frames", loc="left")
    ax_scatter.text(
        0.04,
        0.94,
        f"correlation = {result['heldout_c']['ridge']['pearson_correlation']:.3f}\n"
        f"MSE skill = {100 * result['heldout_c']['ridge']['relative_mse_skill_vs_zero']:.1f}%",
        transform=ax_scatter.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=.4", "facecolor": "white", "edgecolor": GRID, "alpha": 0.9},
    )
    ax_scatter.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "S0 · two sparse simultaneous density diagnostics reconstruct a separated held-out region",
        x=0.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    save(
        fig,
        output_dir,
        "s0_spatial_reconstruction_hero",
        "Old-85604 simultaneous spatial reconstruction from synthetic density diagnostics A and B into held-out C.",
    )


def build_scale_summary(result_dir: Path, output_dir: Path) -> None:
    distance = read_csv(result_dir / "distance_skill.csv")
    modes = read_csv(result_dir / "mode_skill.csv")
    bands = read_csv(result_dir / "mode_band_skill.csv")
    regions = read_csv(result_dir / "region_skill.csv")
    heldout = read_csv(result_dir / "heldout_c_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)

    ridge_c = next(row for row in heldout if row["method"] == "ridge")
    zero_c = next(row for row in heldout if row["method"] == "zero_fluctuation")
    axes[0, 0].bar(
        ["no fluctuation", "ridge from A + B"],
        [float(zero_c["nrmse"]), float(ridge_c["nrmse"])],
        color=["#aab3ba", TEAL],
        width=0.62,
    )
    axes[0, 0].axhline(1.0, color=INK, linewidth=1, linestyle="--")
    axes[0, 0].set_ylabel("held-out C normalized RMSE")
    axes[0, 0].set_ylim(0.0, 1.2)
    axes[0, 0].set_title("A. A separated diagnostic is partially predictable", loc="left")
    axes[0, 0].text(1, float(ridge_c["nrmse"]) + 0.035, f"r = {float(ridge_c['pearson_correlation']):.3f}", ha="center", color=TEAL, fontweight="bold")

    for field in FIELD_COLORS:
        rows = [row for row in distance if row["method"] == "ridge" and row["field"] == field]
        centers = []
        values = []
        for row in rows:
            lo = float(row["lower_m"])
            hi = float(row["upper_m"]) if row["upper_m"] != "inf" else 0.75
            centers.append((lo + hi) / 2.0)
            values.append(float(row["pearson_correlation"]))
        axes[0, 1].plot(centers, values, "o-", color=FIELD_COLORS[field], linewidth=1.8, markersize=4, label=FIELD_LABELS[field])
    axes[0, 1].set_xlabel("distance from observed A/B footprints (m)")
    axes[0, 1].set_ylabel("reconstruction correlation")
    axes[0, 1].set_ylim(0.0, 0.8)
    axes[0, 1].set_title("B. Skill weakens but remains nonzero with distance", loc="left")
    axes[0, 1].legend(ncol=2, fontsize=8)

    for field in FIELD_COLORS:
        rows = [row for row in modes if row["method"] == "ridge" and row["field"] == field and int(row["physical_n"]) <= 100]
        n = [int(row["physical_n"]) for row in rows]
        ratio = [float(row["retained_power_ratio"]) for row in rows]
        axes[1, 0].plot(n, ratio, "o-", color=FIELD_COLORS[field], linewidth=1.5, markersize=3.5, label=FIELD_LABELS[field])
    axes[1, 0].axvspan(20, 35, color=GOLD, alpha=0.15, label=r"evaluated $n=20$–$35$")
    axes[1, 0].axhline(1.0, color=INK, linewidth=1, linestyle="--")
    axes[1, 0].set_xlabel("physical toroidal mode n = 5k")
    axes[1, 0].set_ylabel("predicted / true power")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("C. Reconstruction is band-limited, not full turbulence recovery", loc="left")
    axes[1, 0].legend(ncol=2, fontsize=7.5)

    region_order = ["outboard_midplane", "x_point_stencil", "confined_edge", "private_flux", "scrape_off_layer"]
    matrix = np.full((len(FIELD_COLORS), len(region_order)), np.nan)
    for i, field in enumerate(FIELD_COLORS):
        for j, region in enumerate(region_order):
            row = next(item for item in regions if item["method"] == "ridge" and item["field"] == field and item["region"] == region)
            matrix[i, j] = 100.0 * float(row["relative_mse_skill_vs_zero"])
    image = axes[1, 1].imshow(matrix, vmin=0, vmax=60, cmap="YlGnBu", aspect="auto")
    axes[1, 1].set_yticks(np.arange(len(FIELD_COLORS)), [FIELD_LABELS[field] for field in FIELD_COLORS])
    axes[1, 1].set_xticks(
        np.arange(len(region_order)),
        ["outboard\nmidplane", "X-point", "confined\nedge", "private\nflux", "scrape-off\nlayer"],
    )
    axes[1, 1].set_title("D. MSE skill varies by field and plasma region", loc="left")
    axes[1, 1].grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[1, 1].text(j, i, f"{matrix[i, j]:.0f}%", ha="center", va="center", color="white" if matrix[i, j] > 35 else INK, fontsize=8, fontweight="bold")
    colorbar = fig.colorbar(image, ax=axes[1, 1], shrink=0.85)
    colorbar.set_label("MSE reduction vs no-fluctuation baseline")

    band_rows = [row for row in bands if row["method"] == "ridge" and row["band"] == "evaluated_n_20_to_35"]
    range_text = ", ".join(f"{row['field']} {100 * float(row['retained_power_ratio']):.0f}%" for row in band_rows)
    fig.suptitle(
        "S0 · spatial information is real, long-ranged, and strongly scale dependent\n"
        + r"retained power in $n=20$–$35$: "
        + range_text,
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    save(
        fig,
        output_dir,
        "s0_spatial_skill_by_scale",
        "Held-out diagnostic, distance, physical toroidal mode, and regional reconstruction skill for old-85604 S0.",
    )


def main() -> None:
    args = parse_args()
    required = {
        "result.json",
        "footprints.json",
        "hero_frame.npz",
        "validation_diagnostics.npz",
        "heldout_c_metrics.csv",
        "distance_skill.csv",
        "mode_skill.csv",
        "mode_band_skill.csv",
        "region_skill.csv",
    }
    missing = sorted(name for name in required if not (args.result_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"missing S0 artifacts: {missing}")
    configure_style()
    build_hero(args.result_dir, args.output_dir)
    build_scale_summary(args.result_dir, args.output_dir)
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
