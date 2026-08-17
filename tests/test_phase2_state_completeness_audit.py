from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tcv_diagnostics import state_completeness as state  # noqa: E402


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_script(
    "paper0_state_audit", "paper0/tools/audit_85604_state_completeness.py"
)
MERGE = load_script(
    "paper0_state_merge", "paper0/tools/merge_85604_state_completeness_shards.py"
)
MANIFEST_PATH = ROOT / "paper0" / "manifests" / "phase2_85604_state_completeness.json"


class FakeVariable:
    def __init__(self, name: str, manifest: dict) -> None:
        common = manifest["field_inventory"]["expected_common"]
        expected = manifest["field_inventory"]["expected_per_field"][name]
        self.dimensions = tuple(common["dimensions"])
        self.shape = (624, 8, 6, 81)
        self.dtype = np.dtype("float64")
        self.cell_location = common["cell_location"]
        self.time_dimension = common["time_dimension"]
        self.source = expected["source"]
        if expected["species"] is not None:
            self.species = expected["species"]
        self.units = expected["units"]
        self.conversion = expected["conversion"]


class FakeDataset:
    def __init__(self, manifest: dict) -> None:
        self.variables = {
            name: FakeVariable(name, manifest)
            for name in manifest["field_inventory"]["metadata_inventory_fields"]
        }


class StateAuditStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_metadata_inventory_validates_all_eleven_fields(self) -> None:
        result = AUDIT.validate_and_record_field_metadata(
            FakeDataset(self.manifest), self.manifest, path_name="synthetic.nc"
        )
        self.assertEqual(set(result), set(state.INVENTORY_FIELDS))
        self.assertEqual(result["NVe"]["source"], "evolve_momentum")
        self.assertIsNone(result["Vort"]["species"])
        self.assertEqual(result["phi"]["conversion"], 50.0)

    def test_metadata_inventory_rejects_shape_or_source_drift(self) -> None:
        dataset = FakeDataset(self.manifest)
        dataset.variables["NVe"].shape = (623, 8, 6, 81)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            AUDIT.validate_and_record_field_metadata(
                dataset, self.manifest, path_name="synthetic.nc"
            )
        dataset = FakeDataset(self.manifest)
        dataset.variables["NVe"].source = "wrong"
        with self.assertRaisesRegex(ValueError, "source metadata mismatch"):
            AUDIT.validate_and_record_field_metadata(
                dataset, self.manifest, path_name="synthetic.nc"
            )

    def test_path_guard_refuses_blind_or_ambiguous_root(self) -> None:
        AUDIT.verify_development_path(Path("/data/85604"))
        with self.assertRaisesRegex(ValueError, "held-out"):
            AUDIT.verify_development_path(Path("/data/85606"))
        with self.assertRaisesRegex(ValueError, "must identify"):
            AUDIT.verify_development_path(Path("/data/unknown"))

    def test_scope_accounting_respects_processor_y_coordinate(self) -> None:
        coverage = np.zeros((16, 16), dtype=np.int64)
        coverage[0, 0] = 1
        coverage[0, 15] = 1
        counts = AUDIT.expected_scope_counts_for_coverage(
            coverage, frame_count=2, mxsub=4, mysub=2, native_z=3
        )
        self.assertEqual(counts["full_physical_domain"], 2 * 4 * 4 * 3)
        self.assertEqual(counts["target_dependent_rows"], 2 * 4 * 2 * 3)
        self.assertEqual(counts["guard_independent_transport_interior"], 2 * 4 * 2 * 3)

    def test_field_scope_merge_reconstructs_combined_moments(self) -> None:
        first = state.FieldAccumulator()
        second = state.FieldAccumulator()
        a = np.ones((2, 1, 2, 1), dtype=np.float64)
        b = np.full((2, 1, 2, 1), 3.0, dtype=np.float64)
        first.update(a, x0=0, y0=0)
        second.update(b, x0=1, y0=0)
        merged = MERGE.merge_field([first.result(), second.result()])
        full = merged["scopes"]["full_physical_domain"]
        self.assertEqual(full["total_count"], 8)
        self.assertEqual(full["sum"], 16.0)
        self.assertAlmostEqual(full["rms"], np.sqrt(5.0))
        self.assertEqual(full["minimum"]["location_txyz"], [0, 0, 0, 0])
        self.assertEqual(full["maximum"]["location_txyz"], [0, 1, 0, 0])

    def test_closure_scope_merge_recomputes_global_frame_gate(self) -> None:
        scopes = []
        for error in (0.0, 1e-5):
            accumulator = state.ClosureScopeAccumulator(
                frame_count=2, atol=1e-12, rtol=1e-12
            )
            reference = np.ones((2, 1, 1, 1), dtype=np.float64)
            candidate = reference.copy()
            candidate[1, 0, 0, 0] += error
            accumulator.update(
                reference, candidate, x0=0, global_y=np.asarray([0])
            )
            scopes.append(accumulator.result())
        merged = MERGE.merge_closure_scope(scopes, atol=1e-12, rtol=1e-12)
        self.assertEqual(merged["frame_passed"], [True, False])
        self.assertEqual(merged["point_discrepancy_count"], 1)
        self.assertEqual(merged["failed_frame_indices"], [1])

    def test_strict_json_loader_rejects_nonfinite_constants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text('{"bad": NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                MERGE.strict_json_load(path)


if __name__ == "__main__":
    unittest.main()
