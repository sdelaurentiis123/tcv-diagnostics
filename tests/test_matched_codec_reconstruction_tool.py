"""Static and known-answer tests for the matched codec reconstruction tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/evaluate_matched_codec_reconstruction.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("matched_reconstruction", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestMatchedCodecReconstructionTool(unittest.TestCase):
    def test_tool_keeps_blind_and_exact_operator_boundaries(self) -> None:
        text = TOOL.read_text(encoding="utf-8")
        self.assertIn("held_out_85606_read", text)
        self.assertIn("pending_exact_BOUT_elliptic_phi", text)
        self.assertNotIn("85606/", text)
        self.assertNotIn("clip(", text)

    def test_validation_chunks_cannot_cross_sixteen_frame_blocks(self) -> None:
        tool = load_tool()
        self.assertEqual(
            tool._chunk_stop(510, 624, chunk_frames=16, split="validation"),
            512,
        )
        self.assertEqual(
            tool._chunk_stop(512, 624, chunk_frames=16, split="validation"),
            528,
        )

    def test_cuda_device_is_always_index_bearing(self) -> None:
        tool = load_tool()
        self.assertEqual(str(tool.canonical_cuda_device("cuda")), "cuda:0")
        self.assertEqual(str(tool.canonical_cuda_device("cuda:3")), "cuda:3")
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            tool.canonical_cuda_device("cpu")

    def test_candidate_writer_is_complete_and_refuses_overlap(self) -> None:
        tool = load_tool()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.h5"
            writer = tool.CandidateWriter(
                path,
                family="e6b",
                frames=(496,),
                codec="dcae_l20",
                seed=1701,
                checkpoint_sha256="a" * 64,
            )
            fields = {
                name: np.ones((1, *tool.NATIVE_SHAPE), dtype=np.float32)
                for name in ("Ne", "Pe", "Pi", "Vort")
            }
            boundary = np.ones((1, 2, 32), dtype=np.float32)
            model88 = {
                name: np.ones((1, *tool.VOLUME_SHAPE), dtype=np.float32)
                for name in ("Ne", "Pe", "Pi", "NVi")
            }
            writer.write(0, fields, boundary, model88)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                writer.write(0, fields, boundary, model88)
            writer.finish()
            with h5py.File(path, "r") as handle:
                self.assertEqual(handle.attrs["family"], "e6b")
                np.testing.assert_array_equal(
                    handle["coordinates/frame_index"][:], [496]
                )
                np.testing.assert_array_equal(handle["boundary/Bphi"][:], boundary)
                np.testing.assert_array_equal(handle["model88/NVi"][:], model88["NVi"])


if __name__ == "__main__":
    unittest.main()
