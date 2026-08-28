#!/usr/bin/env python3
"""Render the labeled transport-variogram panel from its stored CSV table."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from tcv_diagnostics.physics_first_figures import save_transport_variogram_figure


H1_ORDER = ("truth", "deterministic", "b5_context", "ecrd", "persistent")
H4_ORDER = ("truth", "persistent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    with args.table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(int(row["horizon_frames"]), row["model_key"])].append(row)

    separations: np.ndarray | None = None

    def records(horizon: int, order: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
        nonlocal separations
        result: dict[str, dict[str, np.ndarray]] = {}
        for key in order:
            rows = sorted(grouped[(horizon, key)], key=lambda row: int(row["lag_native_cells"]))
            if not rows:
                raise ValueError(f"missing horizon={horizon}, model={key}")
            current = np.asarray(
                [float(row["mean_toroidal_arc_separation_m"]) for row in rows],
                dtype=np.float64,
            )
            if separations is None:
                separations = current
            elif not np.array_equal(current, separations):
                raise ValueError("physical separations differ across stored curves")
            result[key] = {
                name: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
                for name, column in (
                    ("mean", "mean"),
                    ("lower_2p5", "lower_2p5"),
                    ("upper_97p5", "upper_97p5"),
                )
            }
        return result

    h1 = records(1, H1_ORDER)
    h4 = records(4, H4_ORDER)
    if separations is None:
        raise ValueError("empty variogram table")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_transport_variogram_figure(
        args.output,
        separation_m=separations,
        h1=h1,
        pgl_h4=h4,
    )


if __name__ == "__main__":
    main()
