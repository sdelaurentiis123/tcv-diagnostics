from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "paper0/results/phase2_o1_codec_transport_6891766.json"
RAW_SHA256 = "c8434cfea29fb4fb9bfa3f8e7fb455985aed6885b478513b06b8d6d8214e3df1"
EXECUTED_COMMIT = "47a26e3ad7e7c8c9a216930dbddd3954e1213e60"


class CodecTransportResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_provenance_scope_and_raw_artifact_are_exact(self) -> None:
        result = self.result
        self.assertEqual(
            result["result_type"], "phase2_o1_codec_transport_oracle_compact"
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["scope"]["run_id"], "85604")
        self.assertEqual(result["scope"]["frame_count"], 624)
        self.assertFalse(result["scope"]["shot_85606_accessed"])
        self.assertFalse(result["scope"]["training_performed"])
        self.assertEqual(result["scope"]["zperiod"], 5)
        self.assertEqual(result["scope"]["mode_mapping"], "n=5*k")
        self.assertEqual(result["execution"]["paper0_commit"], EXECUTED_COMMIT)
        self.assertEqual(str(result["execution"]["slurm_job_id"]), "6891766")
        self.assertEqual(result["raw_artifact"]["sha256"], RAW_SHA256)
        self.assertRegex(RAW_SHA256, re.compile(r"^[0-9a-f]{64}$"))

    def test_alignment_geometry_and_shared_truth_gates_pass(self) -> None:
        result = self.result
        self.assertTrue(result["alignment"]["time_exact"])
        self.assertTrue(result["alignment"]["x_coordinates_exact"])
        self.assertTrue(result["alignment"]["y_coordinates_exact"])
        self.assertLess(
            max(result["alignment"]["maximum_per_frame_relative_l2"].values()),
            1.6e-7,
        )
        self.assertEqual(result["geometry"]["strict_face_row_count"], 1783)
        self.assertEqual(result["geometry"]["separatrix_face_row_count"], 16)
        self.assertEqual(result["geometry"]["separatrix_face_left_model_x"], 15)
        self.assertEqual(result["geometry"]["separatrix_y_inclusive"], [8, 23])
        self.assertTrue(result["shared_truth"]["bitwise_digest_identical"])
        self.assertTrue(
            result["shared_truth"][
                "single_evaluation_fed_to_both_codec_comparisons"
            ]
        )

    def test_formal_gate_outcomes_remain_separate(self) -> None:
        f8 = self.result["codec_results"]["f8"]["gate"]
        z44 = self.result["codec_results"]["z44"]["gate"]
        for gate in (f8, z44):
            self.assertEqual(gate["input_alignment"]["status"], "pass")
            self.assertEqual(gate["input_roundtrip"]["status"], "pass")
            self.assertEqual(gate["c5t_state_adequacy"]["status"], "fail")
            self.assertEqual(gate["prior_preliminary"]["status"], "fail")
            self.assertEqual(gate["full_codec_acceptance"]["status"], "fail")
        self.assertEqual(f8["codec_only_transport"]["status"], "fail")
        self.assertEqual(f8["authoritative_transport"]["status"], "fail")
        self.assertEqual(z44["codec_only_transport"]["status"], "pass")
        self.assertEqual(z44["authoritative_transport"]["status"], "pass")

    def test_state_and_resampling_errors_are_numerically_tiny(self) -> None:
        overall = self.result["codec_results"]["f8"]["overall"]["comparisons"]
        state = overall["P0_vs_P1_state_gap"]["quantities"]
        self.assertEqual(
            state["particle"]["strict_faces"]["metrics"]["relative_l2"], 0.0
        )
        self.assertLess(
            state["electron_internal_energy"]["strict_faces"]["metrics"]
            ["relative_l2"],
            5.2e-8,
        )
        self.assertLess(
            state["ion_internal_energy"]["strict_faces"]["metrics"]["relative_l2"],
            5.1e-7,
        )
        roundtrip = overall["P1_vs_P2_input_roundtrip"]["quantities"]
        for quantity in roundtrip.values():
            self.assertLess(
                quantity["strict_faces"]["metrics"]["relative_l2"], 5e-6
            )
            self.assertLess(
                quantity["separatrix"]["metrics"]["relative_l2"], 3.1e-7
            )

    def test_local_face_and_integrated_surface_conclusions_are_not_conflated(self) -> None:
        results = self.result["codec_results"]
        expected_face = {
            "f8": {
                "particle": 0.2881744200706957,
                "electron_internal_energy": 0.3049476306890191,
                "ion_internal_energy": 0.29402861555971,
                "total_internal_energy": 0.29924730783096304,
            },
            "z44": {
                "particle": 0.2018691244389998,
                "electron_internal_energy": 0.22322352033865597,
                "ion_internal_energy": 0.20924659577283566,
                "total_internal_energy": 0.21596681553538608,
            },
        }
        for codec_name, quantities in expected_face.items():
            comparison = results[codec_name]["overall"]["comparisons"][
                "P0_vs_R_authoritative"
            ]["quantities"]
            for quantity, expected in quantities.items():
                self.assertAlmostEqual(
                    comparison[quantity]["strict_faces"]["metrics"]["relative_l2"],
                    expected,
                )
                self.assertLess(
                    comparison[quantity]["separatrix"]["metrics"]["relative_l2"],
                    0.09,
                )
                self.assertEqual(
                    comparison[quantity]["strict_faces"]["metrics"]["point_count"],
                    90_119_952,
                )
        f8_particle_surface = results["f8"]["overall"]["comparisons"][
            "P0_vs_R_authoritative"
        ]["quantities"]["particle"]["separatrix"]["metrics"]["relative_l2"]
        z44_particle_surface = results["z44"]["overall"]["comparisons"][
            "P0_vs_R_authoritative"
        ]["quantities"]["particle"]["separatrix"]["metrics"]["relative_l2"]
        self.assertLess(f8_particle_surface, z44_particle_surface)

    def test_surface_curves_are_figure_complete_and_block_stable(self) -> None:
        series = self.result["surface_series_si"]
        self.assertEqual(series["units"]["particle"], "s^-1")
        self.assertEqual(series["units"]["total_internal_energy"], "W")
        for quantity, values in series["truth_P0"].items():
            self.assertEqual(len(values), 624, quantity)
            for codec_name in ("f8", "z44"):
                self.assertEqual(
                    len(series["reconstruction_R"][codec_name][quantity]), 624
                )
        for codec_name in ("f8", "z44"):
            blocks = self.result["codec_results"][codec_name]["temporal_blocks"]
            self.assertEqual(len(blocks), 8)
            self.assertEqual(
                [(block["start_inclusive"], block["stop_exclusive"]) for block in blocks],
                [(78 * index, 78 * (index + 1)) for index in range(8)],
            )
            gate = self.result["codec_results"][codec_name]["gate"]
            for gate_name in ("codec_only_transport", "authoritative_transport"):
                for quantity in gate[gate_name]["quantities"].values():
                    self.assertEqual(
                        quantity["temporal_blocks"]["passing_blocks"], 8
                    )

    def test_compact_result_size_is_bounded(self) -> None:
        self.assertLess(RESULT_PATH.stat().st_size, 1_500_000)


if __name__ == "__main__":
    unittest.main()
