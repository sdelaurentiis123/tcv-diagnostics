from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper0/figures/phase2_o1_transport"
PLOTTER = ROOT / "paper0/tools/plot_codec_transport_oracle.py"


class CodecTransportFigureTests(unittest.TestCase):
    def test_all_three_figure_pairs_exist_and_are_nonempty(self) -> None:
        stems = (
            "codec-transport-attribution",
            "codec-separatrix-transport-time-series",
            "codec-transport-error-ladder",
        )
        self.assertEqual(
            {path.name for path in FIGURE_DIR.iterdir()},
            {f"{stem}.{suffix}" for stem in stems for suffix in ("svg", "png")},
        )
        for stem in stems:
            self.assertGreater((FIGURE_DIR / f"{stem}.svg").stat().st_size, 20_000)
            self.assertGreater((FIGURE_DIR / f"{stem}.png").stat().st_size, 50_000)

    def test_svg_labels_define_reduction_units_and_attribution_paths(self) -> None:
        attribution = (FIGURE_DIR / "codec-transport-attribution.svg").read_text(
            encoding="utf-8"
        )
        time_series = (
            FIGURE_DIR / "codec-separatrix-transport-time-series.svg"
        ).read_text(encoding="utf-8")
        ladder = (FIGURE_DIR / "codec-transport-error-ladder.svg").read_text(
            encoding="utf-8"
        )
        for phrase in (
            "All strict physical face contributions",
            "Integrated confined-separatrix wedge flow",
            "Relative L2 error",
        ):
            self.assertIn(phrase, attribution)
        self.assertIn("Time from first stored frame", time_series)
        self.assertIn("Outward wedge flow", time_series)
        self.assertIn("No smoothing", time_series)
        for phrase in ("C5T state", "88→81 input", "end-to-end codec"):
            self.assertIn(phrase, ladder)

    def test_plotter_is_bound_to_the_compact_85604_result_schema(self) -> None:
        source = PLOTTER.read_text(encoding="utf-8")
        self.assertIn("phase2_o1_codec_transport_oracle_compact", source)
        self.assertIn('run_id") != "85604"', source)
        self.assertIn('training_performed") is not False', source)
        self.assertIn("CADENCE_MICROSECONDS = 3.131905426352636", source)


if __name__ == "__main__":
    unittest.main()
