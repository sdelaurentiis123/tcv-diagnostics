from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "paper0/protocol/PHASE2_POTENTIAL_VORTICITY_ALL_FRAME_PROTOCOL.md"
)
MANIFEST = (
    ROOT
    / "paper0/manifests/phase2_potential_vorticity_all_frame_85604.json"
)
PROTOCOL_SHA256 = (
    "b52816365f78755c3433c56af6b71931805a839d9958bdd711e91c4b0b378723"
)
MANIFEST_SHA256 = (
    "a8adb03a5ea6633b114fdb2442739c928f2cb846da9c4a6de43b6355900ef333"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PotentialVorticityAllFrameProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.protocol = PROTOCOL.read_text(encoding="utf-8")

    def test_protocol_and_manifest_are_frozen_by_hash(self) -> None:
        self.assertEqual(sha256(PROTOCOL), PROTOCOL_SHA256)
        self.assertEqual(sha256(MANIFEST), MANIFEST_SHA256)
        self.assertEqual(
            self.manifest["protocol"]["sha256"], PROTOCOL_SHA256
        )

    def test_scope_is_all_85604_frames_and_not_training(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["development_run"], "85604")
        self.assertEqual(manifest["sequestered_run"], "85606")
        self.assertFalse(manifest["held_out_85606_access_allowed"])
        self.assertFalse(manifest["training_allowed"])
        scope = manifest["frame_scope"]
        self.assertEqual(scope["first_index"], 0)
        self.assertEqual(scope["last_index"], 623)
        self.assertEqual(scope["frame_count"], 624)
        self.assertEqual(scope["shape_xyz"], [64, 32, 81])
        self.assertEqual(scope["points_per_field"], 103_514_112)
        self.assertFalse(scope["selection_uses_field_values"])
        self.assertTrue(
            manifest["prior_reads"]["all_frame_85604_values_already_inspected"]
        )

    def test_shards_cover_every_frame_once_and_run_sequentially(self) -> None:
        shards = self.manifest["shards"]
        self.assertEqual(shards["count"], 8)
        self.assertEqual(shards["frames_per_shard"], 78)
        intervals = [tuple(interval) for interval in shards["half_open_intervals"]]
        self.assertEqual(
            intervals,
            [
                (0, 78),
                (78, 156),
                (156, 234),
                (234, 312),
                (312, 390),
                (390, 468),
                (468, 546),
                (546, 624),
            ],
        )
        covered = [frame for start, stop in intervals for frame in range(start, stop)]
        self.assertEqual(covered, list(range(624)))
        self.assertTrue(shards["must_cover_each_frame_exactly_once"])
        self.assertEqual(shards["maximum_concurrent_replays"], 1)
        self.assertEqual(shards["raw_rank_archive_traversals"], 1)

    def test_extraction_is_streamed_and_preserves_boundary_state(self) -> None:
        extraction = self.manifest["canonical_extraction"]
        self.assertEqual(
            extraction["volume_fields"], ["Ne", "Pe", "Pi", "Vort", "phi"]
        )
        self.assertEqual(extraction["volume_shape_per_shard"], [78, 64, 32, 81])
        self.assertEqual(extraction["boundary_shape_per_shard"], [78, 2, 32])
        self.assertEqual(extraction["boundary_side_order"], ["inner", "outer"])
        self.assertTrue(extraction["stream_rank_files_into_shards"])
        self.assertFalse(extraction["hold_complete_624_frame_field_in_memory"])
        self.assertTrue(extraction["preserve_radial_phi_boundary_state"])
        self.assertTrue(extraction["refuse_overwrite"])

    def test_runtime_pressure_inventory_and_equation_are_exact(self) -> None:
        inventory = self.manifest["raw_pressure_identity"]
        self.assertEqual(inventory["negative_raw_Pe_count"], 0)
        self.assertEqual(inventory["negative_raw_Pi_count"], 3_412)
        self.assertEqual(
            inventory["negative_raw_Pi_count_by_shard"],
            [0, 116, 1812, 86, 67, 69, 1262, 0],
        )
        self.assertEqual(
            inventory["minimum_raw_Pi_location_txyz"], [223, 7, 31, 74]
        )
        runtime = self.manifest["runtime_pressure"]
        self.assertEqual(runtime["density_floor"], 1e-7)
        self.assertEqual(runtime["Pi_hat"], "Pi_runtime-Pe_runtime/3672")
        self.assertEqual(runtime["atol"], 1e-12)
        self.assertEqual(runtime["rtol"], 1e-12)
        equation = self.manifest["forward_equation"]
        self.assertEqual(equation["u"], "phi+Pi_hat")
        self.assertEqual(equation["C"], "2/Bxy^2")
        self.assertEqual(equation["forward_vorticity"], "C*L_C(u)")
        self.assertFalse(equation["alternative_relax_potential_fv_operator_allowed"])
        self.assertEqual(equation["toroidal_mode_mapping"], "n=5k")

    def test_ordered_gate_cannot_be_rescued_by_pooled_metrics(self) -> None:
        self.assertEqual(
            self.manifest["ordered_gates"],
            [
                "G0_provenance_and_extraction",
                "G1_compiled_known_answers",
                "G2_compiled_input_and_runtime_pressure",
                "G3_source_forward_closure",
                "G4_exact_eight_shard_merge",
            ],
        )
        gate = self.manifest["source_forward_gate"]
        self.assertEqual(gate["atol"], 5e-10)
        self.assertEqual(gate["rtol"], 5e-10)
        self.assertTrue(gate["all_624_frames_must_pass"])
        self.assertFalse(gate["pooled_metric_can_override"])
        self.assertFalse(gate["correlation_can_override"])
        self.assertFalse(gate["regional_subset_can_override"])
        self.assertFalse(gate["alignment_can_override"])

    def test_every_local_predecessor_lock_matches(self) -> None:
        for name, lock in self.manifest["provenance_locks"].items():
            path = ROOT / lock["path"]
            with self.subTest(name=name):
                self.assertTrue(path.is_file())
                self.assertEqual(sha256(path), lock["sha256"])

    def test_execution_is_single_stream_cpu_rocky9_and_fail_closed(self) -> None:
        execution = self.manifest["execution"]
        self.assertEqual(execution["os_major"], 9)
        self.assertTrue(execution["cpu_only"])
        self.assertEqual(execution["mpi_ranks"], 4)
        self.assertLessEqual(execution["memory_gib_max"], 64)
        self.assertLessEqual(execution["time_limit_minutes_max"], 60)
        self.assertTrue(execution["compile_once_per_top_level_run"])
        self.assertTrue(execution["sequential_shard_replays"])
        self.assertFalse(execution["gpu_requested"])
        decisions = self.manifest["decision_rules"]
        self.assertFalse(decisions["automatic_state_change_authorized"])
        self.assertFalse(decisions["automatic_training_authorized"])
        self.assertFalse(decisions["automatic_held_out_access_authorized"])
        self.assertFalse(decisions["establishes_predictive_sufficiency"])
        self.assertFalse(decisions["establishes_stationarity"])

    def test_human_protocol_defines_math_scope_and_limits(self) -> None:
        for text in (
            "all-624-frame forward-closure calculation",
            "P_s^{\\mathrm{runtime}}",
            "\\mathrm{Vort}_{\\mathrm{forward}}",
            "n=5k",
            "Every one of the 624 frames must pass",
            "never run shards concurrently",
            "does not train a model",
            "do not access 85606",
        ):
            self.assertIn(text, self.protocol)


if __name__ == "__main__":
    unittest.main()
