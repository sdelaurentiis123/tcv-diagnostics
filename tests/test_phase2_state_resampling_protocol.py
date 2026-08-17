from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper0" / "protocol" / "PHASE2_STATE_RESAMPLING_PROTOCOL.md"
MANIFEST = (
    ROOT / "paper0" / "manifests" / "phase2_85604_resampling_sensitivity.json"
)


class StateResamplingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.protocol = PROTOCOL.read_text(encoding="utf-8")

    def test_scope_is_development_only(self) -> None:
        self.assertEqual(self.manifest["development_run"], "85604")
        self.assertEqual(self.manifest["sequestered_runs"], ["85606"])
        self.assertFalse(self.manifest["held_out_85606_access_allowed"])
        for group in ("native_81", "legacy_c5t_88"):
            for source in self.manifest["sources"][group]:
                self.assertIn("85604" if group == "native_81" else "TCV_c5_z88", source["path"])
                self.assertNotIn("85606", source["path"])

    def test_state_policy_preserves_legacy_and_direct_pressure_states(self) -> None:
        policy = self.manifest["state_policy"]
        self.assertEqual(
            policy["legacy_baseline"]["ordered_channels"],
            ["Ne", "Te", "Ti", "phi", "Vi"],
        )
        self.assertEqual(
            policy["transport_candidate"]["ordered_channels"],
            ["Ne", "Pe", "Pi", "phi", "Vi"],
        )
        self.assertFalse(
            policy["transport_candidate"]["complete_markov_state_claimed"]
        )
        self.assertEqual(
            policy["negative_Pi_target_policy"],
            "preserve_direct_evolved_value_without_clipping",
        )
        self.assertEqual(policy["derived_Ti"], "maximum(Pi,0)/Ne")
        self.assertFalse(policy["absolute_frame_number_model_input"])
        self.assertTrue(policy["relative_lead_recorded"])
        self.assertFalse(policy["normalization_fit_by_this_oracle"])

    def test_native_source_partition_covers_every_frame_once(self) -> None:
        intervals = [
            (source["global_start_inclusive"], source["global_stop_exclusive"])
            for source in self.manifest["sources"]["native_81"]
        ]
        self.assertEqual(intervals, [(0, 500), (500, 624)])
        self.assertTrue(
            all(
                source["file_label_is_not_paper0_split"]
                for source in self.manifest["sources"]["native_81"]
            )
        )
        self.assertEqual(self.manifest["data"]["frame_count"], 624)
        self.assertEqual(self.manifest["data"]["global_frame_indices"], [0, 624])

    def test_source_hashes_and_converter_provenance_are_frozen(self) -> None:
        sources = self.manifest["sources"]
        self.assertEqual(
            [source["sha256"] for source in sources["native_81"]],
            [
                "843f9ae99d08fbcdabce977b53e4f6b49be05641a82a387d100b237224b77777",
                "a17b536856c6b8108c0553c300200e074e41407129e47ef402a4de51882ea1ba",
            ],
        )
        converter = sources["predecessor_converter_evidence"]
        self.assertEqual(
            converter["commit"], "24fdb7df11bad5dc6d7b0436afb938ecd09308e9"
        )
        self.assertEqual(
            converter["git_blob"], "ffbb23f917244e3ed847c2568f038533a6d9df76"
        )
        self.assertFalse(converter["imported_or_modified"])

    def test_toroidal_and_resampling_semantics_are_explicit(self) -> None:
        data = self.manifest["data"]
        self.assertEqual(data["native_shape_per_frame"], [64, 32, 81])
        self.assertEqual(data["resampled_shape_per_frame"], [64, 32, 88])
        self.assertEqual(data["axis_order"], ["x", "y", "z"])
        self.assertEqual(data["zperiod"], 5)
        self.assertEqual(data["mode_mapping"], "n=5*k")
        self.assertEqual(data["native_nonnegative_k"], [0, 40])
        self.assertEqual(data["resampled_padding_k"], [41, 44])
        resampling = self.manifest["resampling"]
        self.assertEqual(resampling["implementation"], "scipy.signal.resample")
        self.assertEqual(
            resampling["forward_call"],
            "resample(x,88,axis=-1,window=None,domain='time')",
        )
        self.assertEqual(resampling["output_cast"], "float32")
        self.assertIsNone(resampling["window"])
        self.assertFalse(resampling["smoothing"])

    def test_operator_paths_and_safe_scope_are_frozen(self) -> None:
        self.assertEqual(
            self.manifest["paths_compared"],
            {
                "native": "Q_81(x_81)",
                "round_trip": "Q_81(D(U(x_81)))",
                "direct_88": "Q_88(U(x_81))",
                "aligned_native_88": "U(Q_81(x_81))",
            },
        )
        transport = self.manifest["transport"]
        self.assertEqual(transport["advected_fields"], ["Ne", "Pe", "Pi"])
        self.assertEqual(transport["safe_left_face_indices_inclusive"], [1, 61])
        self.assertEqual(
            transport["safe_divergence_cell_indices_inclusive"], [2, 61]
        )
        self.assertEqual(transport["safe_y_indices_inclusive"], [1, 30])
        self.assertFalse(transport["surface_integrated_physical_flux_claimed"])
        self.assertFalse(transport["si_transport_claimed"])

    def test_gates_precede_implementation(self) -> None:
        observations = self.manifest["protocol_design_observations"]
        self.assertTrue(observations["observed_before_freeze"])
        self.assertTrue(
            observations["selected_raw_equals_native_after_float32_cast"]["passed"]
        )
        self.assertTrue(
            observations["selected_legacy_c5t_resampling_bitwise_exact"]["passed"]
        )
        self.assertFalse(observations["round_trip_field_metric_observed"])
        self.assertFalse(observations["raw64_vs_float32_transport_metric_observed"])
        self.assertFalse(observations["direct_88_transport_metric_observed"])
        self.assertEqual(
            self.manifest["quantile_convention"], "numpy.quantile_method_linear"
        )
        gates = self.manifest["acceptance_gates"]
        self.assertEqual(gates["field_round_trip_max_per_frame_relative_l2"], 2e-6)
        self.assertEqual(
            gates["transport_round_trip_max_aggregate_relative_l2"], 1e-4
        )
        self.assertEqual(
            gates["transport_round_trip_max_p99_per_frame_relative_l2"], 1e-3
        )
        self.assertEqual(
            gates["selected_raw64_vs_float32_transport_max_aggregate_relative_l2"],
            1e-5,
        )
        decisions = self.manifest["decision_rules"]
        self.assertIn("downsample_each_88_cell", decisions["round_trip_pass"])
        self.assertFalse(decisions["automatic_architecture_change"])
        self.assertFalse(decisions["automatic_channel_change"])
        self.assertFalse(self.manifest["training_loss_use"])
        self.assertEqual(
            self.manifest["ensemble_policy"],
            "resample_and_compute_nonlinear_transport_member_wise",
        )

    def test_human_protocol_defines_terms_and_limits_claims(self) -> None:
        for required in (
            "C5T",
            "C5P",
            "n=5k",
            "k=41..44",
            "Q_81(D(U(x_81)))",
            "relative L2",
            "weighted sign disagreement",
            "not a flux-surface integral",
            "No transport quantity is computed only from ensemble-mean fields",
        ):
            self.assertIn(required, self.protocol)


if __name__ == "__main__":
    unittest.main()
