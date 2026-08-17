from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.model_data import (
    StreamingMoments,
    apply_moment_transform,
    array_sha256,
    assert_development_path,
    canonical_float64_sha256,
    records_close,
    relative_l2_by_frame,
    source_segments,
    validate_intervals,
    write_strict_json_atomic,
)


def test_parallel_moments_match_direct_population_statistics():
    values = np.asarray([-3.0, -1.0, 2.0, 4.0, 9.0], dtype=np.float64)
    first = StreamingMoments()
    first.update(values[:2])
    second = StreamingMoments()
    second.update(values[2:])
    first.merge(second)
    result = first.finalize()

    assert result["count"] == values.size
    assert result["mean"] == pytest.approx(float(np.mean(values)), rel=1e-15)
    assert result["M2"] == pytest.approx(
        float(np.sum((values - np.mean(values)) ** 2)), rel=1e-15
    )
    assert result["population_variance"] == pytest.approx(
        float(np.var(values)), rel=1e-15
    )
    restored = StreamingMoments.from_record(result)
    assert restored.finalize() == result


def test_moment_record_comparison_requires_exact_count():
    first = StreamingMoments()
    first.update(np.asarray([1.0, 2.0, 3.0]))
    reference = first.finalize()
    nearby = dict(reference)
    nearby["mean"] += 1e-13
    comparisons = records_close(
        reference,
        nearby,
        relative_tolerance=1e-12,
        absolute_tolerance=1e-12,
    )
    assert all(comparisons.values())

    wrong_count = dict(nearby)
    wrong_count["count"] += 1
    assert not records_close(
        reference,
        wrong_count,
        relative_tolerance=1e-12,
        absolute_tolerance=1e-12,
    )["count"]


def test_frozen_transforms_log_only_density_and_preserve_negative_pressure():
    transforms = {
        "Ne": {"name": "log_offset", "offset": 1e-6},
        "Pi": {"name": "identity"},
    }
    density = np.asarray([1.0, 2.0])
    pressure = np.asarray([-2.0, 3.0])
    np.testing.assert_allclose(
        apply_moment_transform("Ne", density, transforms),
        np.log(density + 1e-6),
    )
    np.testing.assert_array_equal(
        apply_moment_transform("Pi", pressure, transforms),
        pressure,
    )
    with pytest.raises(ValueError, match="nonpositive"):
        apply_moment_transform("Ne", np.asarray([-1e-6]), transforms)


def test_source_segments_cross_the_legacy_file_boundary_exactly_once():
    sources = [
        {"global_start_inclusive": 0, "global_stop_exclusive": 500, "path": "a"},
        {
            "global_start_inclusive": 500,
            "global_stop_exclusive": 624,
            "path": "b",
        },
    ]
    segments = source_segments(sources, 468, 546)
    assert [
        (item[0]["path"], *item[1:])
        for item in segments
    ] == [
        ("a", 468, 500, 468, 500),
        ("b", 0, 46, 500, 546),
    ]
    with pytest.raises(ValueError, match="exactly once"):
        source_segments(sources, 600, 625)


def test_interval_and_held_out_guards_fail_closed():
    assert validate_intervals(
        [[0, 2], [2, 5]], expected_start=0, expected_stop=5
    ) == ((0, 2), (2, 5))
    with pytest.raises(ValueError, match="exact ordered coverage"):
        validate_intervals([[0, 2], [3, 5]], expected_start=0, expected_stop=5)
    assert_development_path(Path("/tmp/TCV_85604/data.h5"))
    with pytest.raises(ValueError, match="held-out"):
        assert_development_path(Path("/tmp/TCV_85606/data.h5"))


def test_array_hash_locks_shape_dtype_and_float64_extraction_semantics():
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert array_sha256(values) == array_sha256(values.copy())
    assert array_sha256(values) != array_sha256(values.astype(np.float64))
    assert array_sha256(values) != array_sha256(values.reshape(3, 2))

    float64 = values.astype(np.float64)
    assert canonical_float64_sha256(float64) == array_sha256(float64)


def test_relative_l2_is_framewise_and_defines_zero_norm_case():
    reference = np.asarray([[3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    candidate = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    result = relative_l2_by_frame(reference, candidate)
    np.testing.assert_array_equal(result[:2], [1.0, 0.0])
    assert np.isinf(result[2])


def test_strict_atomic_json_refuses_nonfinite_and_overwrite():
    with TemporaryDirectory() as directory:
        output = Path(directory) / "result.json"
        write_strict_json_atomic(output, {"value": 1.0})
        assert output.read_text().endswith("\n")
        with pytest.raises(FileExistsError):
            write_strict_json_atomic(output, {"value": 2.0})
        with pytest.raises(ValueError):
            write_strict_json_atomic(Path(directory) / "nan.json", {"x": np.nan})
