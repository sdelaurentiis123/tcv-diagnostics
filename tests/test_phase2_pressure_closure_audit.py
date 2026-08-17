from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper0" / "tools"
AUDITOR = TOOLS / "audit_85604_pressure_closure.py"
MERGER = TOOLS / "merge_85604_pressure_closure_shards.py"
MANIFEST = (
    ROOT / "paper0" / "manifests" / "phase2_85604_pressure_closure_audit.json"
)
LAUNCHER = ROOT / "cluster" / "phase2_85604_pressure_closure_audit.sbatch"
NO_RESULT = ROOT / "paper0" / "results" / "phase2_pressure_closure_6891417.json"
PARALLEL_NO_RESULT = (
    ROOT / "paper0" / "results" / "phase2_pressure_closure_6891530.json"
)
MEMORY_NO_RESULT = (
    ROOT / "paper0" / "results" / "phase2_pressure_closure_6891570.json"
)
PREEMPTED_NO_RESULT = (
    ROOT / "paper0" / "results" / "phase2_pressure_closure_6891571.json"
)
COMPLETE_RESULT = (
    ROOT / "paper0" / "results" / "phase2_pressure_closure_6891583.json"
)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
AUDIT = load_module("paper0_pressure_closure_audit", AUDITOR)
MERGE = load_module("paper0_pressure_closure_merge", MERGER)


class PressureClosureAuditImplementationTests(unittest.TestCase):
    def test_launcher_is_cpu_only_clean_locked_and_syntax_valid(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        auditor_digest = hashlib.sha256(AUDITOR.read_bytes()).hexdigest()
        merger_digest = hashlib.sha256(MERGER.read_bytes()).hexdigest()
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "--ntasks=1",
            "--cpus-per-task=16",
            "--no-requeue",
            "Refusing to overwrite",
            manifest_digest,
            auditor_digest,
            merger_digest,
            "920ba829cc78cdab0dbf6101c69fecc4689bd8dd",
            "audit_85604_pressure_closure.py",
            "merge_85604_pressure_closure_shards.py",
            "pressure_closure_audit.json",
            "SHARD_COUNT=16",
            "--shard-count",
            "srun",
            "--exclusive",
            "--exact",
            "--mem=4G",
        ):
            self.assertIn(required, text)
        self.assertNotIn("--gres=gpu", text)
        self.assertNotIn("85606", text)

    def test_scope_selection_uses_declared_global_y(self) -> None:
        np.testing.assert_array_equal(
            AUDIT.scope_y_indices(np.array([0, 1]), "full_physical_domain"),
            np.array([0, 1]),
        )
        np.testing.assert_array_equal(
            AUDIT.scope_y_indices(
                np.array([0, 1]), "guard_independent_transport_interior"
            ),
            np.array([1]),
        )

    def test_partial_scope_accounting_respects_target_rows(self) -> None:
        coverage = np.zeros((2, 2), dtype=np.int64)
        coverage[0, 0] = 1
        counts = AUDIT.expected_scope_counts_for_coverage(
            coverage,
            frame_count=2,
            mxsub=1,
            mysub=2,
            native_z=1,
        )
        self.assertEqual(counts["full_physical_domain"], 4)
        self.assertEqual(counts["guard_independent_transport_interior"], 2)
        self.assertEqual(counts["target_dependent_rows"], 2)
        np.testing.assert_array_equal(
            AUDIT.scope_y_indices(np.array([30, 31]), "target_dependent_rows"),
            np.array([1]),
        )

    def test_value_accumulator_localizes_negative_target_and_interior_points(self) -> None:
        values = np.ones((2, 2, 2, 2), dtype=np.float64)
        values[0, 0, 0, 0] = -3.0
        values[1, 1, 1, 1] = -2.0
        accumulator = AUDIT.ValueAccumulator(
            frame_count=2,
            nx=2,
            ny=2,
            temporal_blocks=[(0, 0), (1, 1)],
        )
        accumulator.update(values, x0=0, y0=0)
        result = accumulator.result()
        full = result["scopes"]["full_physical_domain"]
        interior = result["scopes"]["guard_independent_transport_interior"]
        targets = result["scopes"]["target_dependent_rows"]
        self.assertEqual(full["negative_count"], 2)
        self.assertEqual(interior["negative_count"], 1)
        self.assertEqual(targets["negative_count"], 1)
        self.assertEqual(full["minimum"]["location_txyz"], [0, 0, 0, 0])
        self.assertEqual(result["negative_count_by_frame"], [1, 1])
        self.assertEqual(result["negative_count_by_y"], [1, 1])
        self.assertEqual(result["negative_count_by_temporal_block"], [1, 1])

    def test_closure_accumulator_separates_frame_and_point_rules(self) -> None:
        reference = np.ones((2, 2, 2, 2), dtype=np.float64)
        candidate = reference.copy()
        reference[1, 0, 0, 1] = -1.0e-4
        candidate[1, 0, 0, 1] = 0.0
        accumulator = AUDIT.ClosureAccumulator(
            frame_count=2,
            nx=2,
            ny=2,
            temporal_blocks=[(0, 0), (1, 1)],
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        accumulator.update(reference, candidate, x0=0, y0=0)
        result = accumulator.result()
        full = result["scopes"]["full_physical_domain"]
        interior = result["scopes"]["guard_independent_transport_interior"]
        targets = result["scopes"]["target_dependent_rows"]
        self.assertEqual(full["frame_fail_count"], 1)
        self.assertEqual(full["failed_frame_indices"], [1])
        self.assertEqual(full["point_discrepancy_count"], 1)
        self.assertEqual(full["negative_reference_discrepancy_count"], 1)
        self.assertEqual(interior["point_discrepancy_count"], 0)
        self.assertEqual(targets["point_discrepancy_count"], 1)
        self.assertEqual(full["maximum_error"]["location_txyz"], [1, 0, 0, 1])

    def test_relation_candidates_use_direct_pressure_references(self) -> None:
        fields = {
            "Ne": np.array([2.0]),
            "Ni": np.array([2.0]),
            "Te": np.array([3.0]),
            "Ti": np.array([4.0]),
            "Pe": np.array([6.0]),
            "Pi": np.array([8.0]),
        }
        reference, candidate = AUDIT.relation_candidate(
            "Pi_equals_Ne_times_Ti", fields
        )
        np.testing.assert_array_equal(reference, fields["Pi"])
        np.testing.assert_array_equal(candidate, np.array([8.0]))

    def test_stream_digest_locks_rank_coordinates_and_values(self) -> None:
        first = AUDIT.initialize_stream_digests(["Pe"], [1, 1, 1, 2])["Pe"]
        second = AUDIT.initialize_stream_digests(["Pe"], [1, 1, 1, 2])["Pe"]
        values = np.array([[[[1.0, 2.0]]]])
        AUDIT.update_stream_digest(first, values, rank=0, pe_x=0, pe_y=0)
        AUDIT.update_stream_digest(second, values, rank=1, pe_x=0, pe_y=0)
        self.assertNotEqual(first.hexdigest(), second.hexdigest())

    def test_value_shard_merge_matches_single_pass(self) -> None:
        temporal_blocks = [(0, 0), (1, 1)]
        values = np.ones((2, 2, 2, 1), dtype=np.float64)
        values[0, 0, 0, 0] = -3.0
        values[1, 1, 1, 0] = -2.0
        complete = AUDIT.ValueAccumulator(
            frame_count=2, nx=2, ny=2, temporal_blocks=temporal_blocks
        )
        complete.update(values, x0=0, y0=0)
        partials = []
        for x_index in range(2):
            partial = AUDIT.ValueAccumulator(
                frame_count=2, nx=2, ny=2, temporal_blocks=temporal_blocks
            )
            partial.update(
                values[:, x_index : x_index + 1], x0=x_index, y0=0
            )
            partials.append(partial.result())
        merged = MERGE.merge_value_field(partials, temporal_blocks)
        self.assertEqual(merged, complete.result())

    def test_closure_shard_merge_matches_single_pass(self) -> None:
        temporal_blocks = [(0, 0), (1, 1)]
        reference = np.ones((2, 2, 2, 1), dtype=np.float64)
        candidate = reference.copy()
        reference[1, 1, 0, 0] = -1.0e-4
        candidate[1, 1, 0, 0] = 0.0
        kwargs = {
            "frame_count": 2,
            "nx": 2,
            "ny": 2,
            "temporal_blocks": temporal_blocks,
            "atol": 1.0e-12,
            "rtol": 1.0e-12,
        }
        complete = AUDIT.ClosureAccumulator(**kwargs)
        complete.update(reference, candidate, x0=0, y0=0)
        partials = []
        for x_index in range(2):
            partial = AUDIT.ClosureAccumulator(**kwargs)
            partial.update(
                reference[:, x_index : x_index + 1],
                candidate[:, x_index : x_index + 1],
                x0=x_index,
                y0=0,
            )
            partials.append(partial.result())
        merged = MERGE.merge_closure_relation(
            partials,
            temporal_blocks=temporal_blocks,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        self.assertEqual(merged, complete.result())

    def test_digest_tree_is_order_sensitive_and_repeatable(self) -> None:
        first = [{"shard": 0, "stream_sha256": "a" * 64}]
        second = [{"shard": 1, "stream_sha256": "a" * 64}]
        self.assertEqual(MERGE.digest_tree(first), MERGE.digest_tree(first))
        self.assertNotEqual(MERGE.digest_tree(first), MERGE.digest_tree(second))

    def test_strict_json_writer_refuses_overwrite_and_nonfinite(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            AUDIT.strict_json_write(output, {"valid": 1.0})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"valid": 1.0})
            with self.assertRaises(FileExistsError):
                AUDIT.strict_json_write(output, {"valid": 2.0})
            invalid = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                AUDIT.strict_json_write(invalid, {"bad": float("nan")})

    def test_held_out_path_guard_is_component_exact(self) -> None:
        AUDIT._verify_path_is_development_run(Path("/tmp/data/85604"))
        with self.assertRaises(ValueError):
            AUDIT._verify_path_is_development_run(Path("/tmp/data/85606"))
        with self.assertRaises(ValueError):
            AUDIT._verify_path_is_development_run(Path("/tmp/data/unknown"))

    def test_serial_attempt_is_tracked_as_no_result(self) -> None:
        attempt = json.loads(NO_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(attempt["paper0_commit"], "39bfb22ebd2eed9ee67bc193d958298857fd1e21")
        self.assertEqual(attempt["slurm"]["job_id"], 6891417)
        self.assertEqual(attempt["slurm"]["state"], "CANCELLED")
        self.assertFalse(attempt["data_access"]["complete_rank_coverage_established"])
        self.assertFalse(attempt["data_access"]["scientific_result_written"])
        self.assertFalse(attempt["outcome"]["scientific_statistics_accepted"])
        self.assertFalse(attempt["outcome"]["protocol_or_tolerance_changed"])
        self.assertFalse(attempt["data_access"]["held_out_85606_read"])

    def test_first_parallel_attempt_is_tracked_as_no_result(self) -> None:
        attempt = json.loads(PARALLEL_NO_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(attempt["paper0_commit"], "b672d69bcadcf411ddf9549a4e52a23294d5d0f3")
        self.assertEqual(attempt["slurm"]["job_id"], 6891530)
        self.assertEqual(attempt["slurm"]["started_shard_step_count"], 1)
        self.assertEqual(attempt["data_access"]["partial_json_count"], 0)
        self.assertFalse(attempt["data_access"]["scientific_result_written"])
        self.assertFalse(attempt["outcome"]["scientific_statistics_accepted"])
        self.assertFalse(attempt["outcome"]["protocol_or_tolerance_changed"])
        self.assertFalse(attempt["data_access"]["held_out_85606_read"])

    def test_parallel_memory_attempt_is_tracked_as_no_result(self) -> None:
        attempt = json.loads(MEMORY_NO_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(attempt["paper0_commit"], "347495f0f4f09461dc47567331df31df7f785ca8")
        self.assertEqual(attempt["slurm"]["job_id"], 6891570)
        self.assertEqual(attempt["outcome"]["observation"]["step_zero_tres"], "cpu=1,mem=64G,node=1")
        self.assertEqual(attempt["data_access"]["partial_json_count"], 0)
        self.assertFalse(attempt["data_access"]["scientific_result_written"])
        self.assertFalse(attempt["outcome"]["scientific_statistics_accepted"])
        self.assertFalse(attempt["outcome"]["protocol_or_tolerance_changed"])
        self.assertFalse(attempt["data_access"]["held_out_85606_read"])

    def test_preempted_parallel_attempt_is_tracked_as_no_result(self) -> None:
        attempt = json.loads(PREEMPTED_NO_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(attempt["paper0_commit"], "f5d4541e7f048a43948ba00d6fd828e0d05141e9")
        self.assertEqual(attempt["slurm"]["job_id"], 6891571)
        self.assertEqual(attempt["slurm"]["state"], "PREEMPTED")
        self.assertEqual(attempt["slurm"]["started_shard_step_count"], 16)
        self.assertEqual(attempt["data_access"]["partial_json_count"], 0)
        self.assertFalse(attempt["data_access"]["scientific_result_written"])
        self.assertFalse(attempt["outcome"]["scientific_statistics_accepted"])
        self.assertFalse(attempt["outcome"]["protocol_or_tolerance_changed"])
        self.assertFalse(attempt["data_access"]["held_out_85606_read"])

    def test_complete_result_records_frozen_decision(self) -> None:
        result = json.loads(COMPLETE_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(result["paper0_commit"], "f5d4541e7f048a43948ba00d6fd828e0d05141e9")
        self.assertEqual(result["slurm"]["job_id"], 6891583)
        self.assertEqual(result["slurm"]["state"], "COMPLETED")
        self.assertEqual(result["slurm"]["partition"], "gen")
        self.assertFalse(result["data_scope"]["held_out_85606_read"])
        self.assertEqual(result["data_scope"]["shape_per_field"], [624, 64, 32, 81])
        self.assertEqual(result["data_scope"]["zperiod"], 5)
        self.assertTrue(result["integrity"]["all_shards_completed_before_merge"])
        self.assertEqual(result["integrity"]["partial_json_count"], 16)
        self.assertEqual(result["value_findings"]["Pi"]["full_negative_count"], 3412)
        self.assertEqual(result["value_findings"]["Pi"]["interior_negative_count"], 1421)
        ion = result["closure_findings"]["Pi_equals_Ne_times_Ti"]
        self.assertEqual(ion["full_failed_frames"], 72)
        self.assertEqual(ion["interior_failed_frames"], 47)
        self.assertEqual(ion["negative_reference_discrepancy_count"], 3412)
        self.assertEqual(ion["nonnegative_reference_discrepancy_count"], 0)
        self.assertFalse(
            result["scientific_findings"]
            ["temperature_state_reproduces_guard_independent_pressure_transport"]
        )
        self.assertFalse(result["scientific_findings"]["automatic_channel_change_authorized"])


if __name__ == "__main__":
    unittest.main()
