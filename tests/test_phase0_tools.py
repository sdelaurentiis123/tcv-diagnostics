from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_module("audit_85604_data", "paper0/tools/audit_85604_data.py")
summarize = load_module(
    "summarize_da_reproduction", "paper0/tools/summarize_da_reproduction.py"
)


def write_raw(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        for name, value in {
            "zperiod": 5,
            "ZMIN": 0.0,
            "ZMAX": 0.2,
            "Omega_ci": 100.0,
            "Bnorm": 1.0,
            "Nnorm": 1e19,
            "Tnorm": 50.0,
            "rho_s0": 7e-4,
            "Cs0": 69000.0,
        }.items():
            handle.create_dataset(name, data=value)
        handle.create_dataset(
            "t_array", data=np.asarray([0.0, 300.0, 600.0, 900.0])
        )
        handle.create_dataset("Ne", shape=(4, 8, 6, 81), dtype=np.float32)


def write_well(path: Path, times: list[float]) -> None:
    fields = ["Ne", "Te", "Ti", "phi", "Vi"]
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        dimensions = handle.create_group("dimensions")
        dimensions.attrs.create(
            "spatial_dims", np.asarray(["x", "y", "z"], dtype=object), dtype=string_dtype
        )
        dimensions.create_dataset("time", data=np.asarray(times, dtype=np.float64))
        dimensions.create_dataset("x", data=np.arange(2, dtype=np.float64))
        dimensions.create_dataset("y", data=np.arange(2, dtype=np.float64))
        dimensions.create_dataset("z", data=np.arange(4, dtype=np.float64))

        boundary = handle.create_group("boundary_conditions")
        for name, size, kind in (
            ("x_wall", 2, "WALL"),
            ("y_wall", 2, "WALL"),
            ("z_periodic", 4, "PERIODIC"),
        ):
            group = boundary.create_group(name)
            group.attrs["bc_type"] = kind
            group.create_dataset("mask", data=np.ones(size, dtype=bool))

        t0 = handle.create_group("t0_fields")
        t0.attrs.create("field_names", np.asarray(fields, dtype=object), dtype=string_dtype)
        for field in fields:
            t0.create_dataset(
                field,
                shape=(1, len(times), 2, 2, 4),
                dtype=np.float32,
            )


class DataAuditTests(unittest.TestCase):
    def test_metadata_and_zero_guard_are_computed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "BOUT.dmp.0.nc"
            train_path = root / "train_85604.hdf5"
            valid_path = root / "valid_85604.hdf5"
            write_raw(raw_path)
            write_well(train_path, [0.0, 300.0])
            write_well(valid_path, [600.0, 900.0])

            raw = audit.audit_raw_bout(raw_path, expected_zperiod=5)
            train = audit.audit_well(train_path)
            valid = audit.audit_well(valid_path)
            boundary = audit.compare_time_regions(raw, train, valid)

            self.assertEqual(raw["zperiod"], 5)
            self.assertEqual(raw["mode_mapping"], "n = 5k")
            self.assertAlmostEqual(raw["frame_cadence_seconds"], 3.0)
            self.assertEqual(train["fields"], ["Ne", "Te", "Ti", "phi", "Vi"])
            self.assertFalse(train["all_field_units_declared"])
            self.assertEqual(boundary["guard_frames"], 0)
            self.assertTrue(boundary["matches_raw_frame_count"])
            self.assertTrue(boundary["matches_raw_endpoints"])

    def test_sequestered_names_are_rejected(self) -> None:
        self.assertFalse(audit._path_is_allowed(Path("/tmp/85606/data.h5")))
        self.assertFalse(audit._path_is_allowed(Path("/tmp/test/data.h5")))
        self.assertTrue(audit._path_is_allowed(Path("/tmp/85604/valid/data.h5")))


def locked_summary(split: str = "valid") -> dict:
    return {
        "dataset_split": split,
        "trajectory_index": 0,
        "trajectory_start": 24,
        "horizon": 48,
        "context": 1,
        "update": "etkf",
        "analysis_mode": "filter",
        "layout": "iter",
        "n_probes": 69,
        "samples": 64,
        "ensemble_M": 64,
        "assim_every": 4,
        "inflate": "off",
        "observation_noise_policy": "frame_keyed_common_random_numbers",
        "inflation": 1.0,
        "obs_std": 0.05,
        "fields": ["Ne", "Te", "Ti", "phi", "Vi"],
        "is_oracle_analysis": False,
        "crn_paired": True,
        "err_free": [9.0, summarize.EXPECTED_FREE, summarize.EXPECTED_FREE],
        "err_anchored": [9.0, summarize.EXPECTED_ANCHORED, summarize.EXPECTED_ANCHORED],
        "mean_err_free": summarize.EXPECTED_FREE,
        "mean_err_anchored": summarize.EXPECTED_ANCHORED,
    }


class ReproductionSummaryTests(unittest.TestCase):
    def invoke(self, directory: Path, summary: dict) -> Path:
        source = directory / "da_summary.json"
        command = directory / "command.sh"
        output = directory / "reproduction_summary.json"
        source.write_text(json.dumps(summary), encoding="utf-8")
        command.write_text("python da_anchored_rollout.py split=valid update=etkf\n")
        argv = [
            "summarize_da_reproduction.py",
            "--summary",
            str(source),
            "--command-file",
            str(command),
            "--output",
            str(output),
            "--paper0-commit",
            "abc123",
            "--slurm-job-id",
            "42",
        ]
        previous = sys.argv
        try:
            sys.argv = argv
            summarize.main()
        finally:
            sys.argv = previous
        return output

    def test_historical_reference_is_reproduced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = self.invoke(Path(directory), locked_summary())
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["reproduction"]["status"], "within_tolerance")
            self.assertGreater(
                result["reproduction"]["absolute_rmse_improvement"], 0.0
            )
            self.assertFalse(result["blind_test_accessed"])

    def test_wrong_split_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
                self.invoke(Path(directory), locked_summary(split="train"))


if __name__ == "__main__":
    unittest.main()
