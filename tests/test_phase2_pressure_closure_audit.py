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
MANIFEST = (
    ROOT / "paper0" / "manifests" / "phase2_85604_pressure_closure_audit.json"
)
LAUNCHER = ROOT / "cluster" / "phase2_85604_pressure_closure_audit.sbatch"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


AUDIT = load_module("paper0_pressure_closure_audit", AUDITOR)


class PressureClosureAuditImplementationTests(unittest.TestCase):
    def test_launcher_is_cpu_only_clean_locked_and_syntax_valid(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        auditor_digest = hashlib.sha256(AUDITOR.read_bytes()).hexdigest()
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "--ntasks=1",
            "--no-requeue",
            "Refusing to overwrite",
            manifest_digest,
            auditor_digest,
            "920ba829cc78cdab0dbf6101c69fecc4689bd8dd",
            "audit_85604_pressure_closure.py",
            "pressure_closure_audit.json",
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


if __name__ == "__main__":
    unittest.main()
