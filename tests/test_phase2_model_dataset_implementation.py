from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "paper0/tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


BUILD = load_module(
    "paper0_build_model_dataset",
    TOOLS / "build_85604_model_dataset_shard.py",
)
MERGE = load_module(
    "paper0_merge_model_dataset",
    TOOLS / "merge_85604_model_dataset_shards.py",
)

from tcv_diagnostics.model_data import VOLUME_FIELDS, array_sha256  # noqa: E402


def write_well(path: Path, values: np.ndarray, time: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        fields = handle.create_group("t0_fields")
        fields.create_dataset("Ne", data=values[None], dtype="f4")
        dimensions = handle.create_group("dimensions")
        dimensions.create_dataset("time", data=time, dtype="f8")


def test_actual_manifest_passes_static_builder_validation():
    manifest_path = ROOT / "paper0/manifests/phase2_model_dataset_85604.json"
    protocol_path = ROOT / "paper0/protocol/PHASE2_MODEL_DATASET_PROTOCOL.md"
    manifest = json.loads(manifest_path.read_text())
    intervals = BUILD.verify_manifest(
        manifest,
        protocol_path=protocol_path,
        shard_index=7,
    )
    assert intervals[7] == (546, 624)


def test_global_well_reader_handles_source_boundary_and_distinct_z_sizes():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        first_path = root / "TCV_85604_first.h5"
        second_path = root / "TCV_85604_second.h5"
        first = np.arange(2 * 64 * 32 * 3, dtype=np.float32).reshape(
            2, 64, 32, 3
        )
        second = (100000 + np.arange(2 * 64 * 32 * 3)).astype(
            np.float32
        ).reshape(2, 64, 32, 3)
        write_well(first_path, first, np.asarray([10.0, 20.0]))
        write_well(second_path, second, np.asarray([30.0, 40.0]))
        sources = [
            {
                "path": str(first_path),
                "global_start_inclusive": 0,
                "global_stop_exclusive": 2,
                "field_shape": [1, 2, 64, 32, 3],
                "field_dataset_template": "t0_fields/{field}",
                "time_dataset": "dimensions/time",
            },
            {
                "path": str(second_path),
                "global_start_inclusive": 2,
                "global_stop_exclusive": 4,
                "field_shape": [1, 2, 64, 32, 3],
                "field_dataset_template": "t0_fields/{field}",
                "time_dataset": "dimensions/time",
            },
        ]
        values = BUILD.read_well_field(
            sources,
            field="Ne",
            start=1,
            stop=3,
            expected_z=3,
        )
        np.testing.assert_array_equal(values, np.stack([first[1], second[0]]))
        np.testing.assert_array_equal(
            BUILD.read_well_time(sources, start=1, stop=3),
            [20.0, 30.0],
        )


def test_source_preflight_requires_exact_path_and_digest_set():
    manifest = json.loads(
        (ROOT / "paper0/manifests/phase2_model_dataset_85604.json").read_text()
    )
    expected = BUILD.required_preflight_hashes(manifest)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "source_sha256.txt"
        path.write_text(
            "".join(
                f"{digest}  {source}\n"
                for source, digest in expected.items()
            )
        )
        assert BUILD.verify_source_preflight(path, manifest) == expected
        path.write_text(path.read_text().replace(next(iter(expected.values())), "0" * 64))
        try:
            BUILD.verify_source_preflight(path, manifest)
        except ValueError as error:
            assert "differs" in str(error)
        else:
            raise AssertionError("mismatched source preflight was accepted")


def test_hdf_writer_echo_locks_every_field_boundary_and_coordinate():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "shard.h5"
        values = {
            field: (
                np.arange(12, dtype=np.float32).reshape(2, 2, 1, 3)
                + index
            )
            for index, field in enumerate(VOLUME_FIELDS)
        }
        boundary = np.arange(128, dtype=np.float32).reshape(2, 2, 32)
        frames = np.asarray([4, 5], dtype=np.int64)
        time = np.asarray([1200.0, 1500.0], dtype=np.float64)
        manifest = {
            "phase": "synthetic_model_dataset",
            "output": {"volume_chunk_shape": [1, 2, 1, 3]},
        }
        BUILD.create_output_file(
            path,
            manifest=manifest,
            manifest_sha256="a" * 64,
            protocol_sha256="b" * 64,
            paper0_commit="c" * 40,
            slurm_job_id=1,
            shard_index=0,
            start=4,
            stop=6,
            frame_index=frames,
            time=time,
            volume_values=values,
            boundary=boundary,
        )
        echo = BUILD.verify_reopened_output(
            path,
            expected_digests={
                field: array_sha256(array) for field, array in values.items()
            },
            expected_boundary_digest=array_sha256(boundary),
            frame_index=frames,
            time=time,
            start=4,
            stop=6,
        )
        assert echo["all_bitwise_exact"]


def test_reducer_moment_recomputation_reads_only_training_frames():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        paths = []
        intervals = ((0, 2), (2, 4))
        for index, (start, stop) in enumerate(intervals):
            path = root / f"shard_{index}.h5"
            with h5py.File(path, "w") as handle:
                fields = handle.create_group("fields")
                values = np.arange(start, stop, dtype=np.float32).reshape(2, 1, 1, 1)
                fields.create_dataset("Pe", data=values)
                boundary = handle.create_group("boundary")
                boundary.create_dataset(
                    "Bphi",
                    data=np.repeat(values.reshape(2, 1, 1), 2, axis=1),
                )
            paths.append(path)
        transforms = {
            "Pe": {"name": "identity"},
            "Bphi": {"name": "identity"},
        }
        record = MERGE.recompute_output_moments(
            paths,
            intervals,
            field="Pe",
            transforms=transforms,
            training_stop=3,
        )
        assert record["count"] == 3
        assert record["mean"] == 1.0
        assert record["population_variance"] == 2.0 / 3.0


def test_normalization_gate_rejects_an_exact_but_constant_channel():
    variable = {
        "count": 3,
        "mean": 1.0,
        "M2": 2.0,
        "population_variance": 2.0 / 3.0,
        "population_standard_deviation": np.sqrt(2.0 / 3.0),
    }
    accepted = MERGE.normalization_comparison(
        variable,
        variable,
        expected_count=3,
        relative_tolerance=1e-12,
        absolute_tolerance=1e-12,
    )
    assert accepted["passed"]
    assert accepted["standard_deviation_strictly_positive"]

    constant = {
        "count": 3,
        "mean": 1.0,
        "M2": 0.0,
        "population_variance": 0.0,
        "population_standard_deviation": 0.0,
    }
    rejected = MERGE.normalization_comparison(
        constant,
        constant,
        expected_count=3,
        relative_tolerance=1e-12,
        absolute_tolerance=1e-12,
    )
    assert not rejected["passed"]
    assert not rejected["standard_deviation_strictly_positive"]
